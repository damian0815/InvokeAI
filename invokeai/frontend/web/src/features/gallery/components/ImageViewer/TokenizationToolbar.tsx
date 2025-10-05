import {
  Button,
  ButtonGroup,
  Flex,
  Icon,
  Kbd,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectTokenizationPrompt } from 'features/gallery/store/gallerySelectors';
import { imageToTokenizeChanged, tokenizationDisplayChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { Trans, useTranslation } from 'react-i18next';
import { PiQuestion } from 'react-icons/pi';

export const TokenizationToolbar = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const tokenizationPrompt = useAppSelector(selectTokenizationPrompt);
  const setTokenizationDisplayPositivePrompt = useCallback(() => {
    dispatch(tokenizationDisplayChanged('positive'));
  }, [dispatch]);
  const setTokenizationDisplayNegativePrompt = useCallback(() => {
    dispatch(tokenizationDisplayChanged('negative'));
  }, [dispatch]);
    const exitTokenization = useCallback(() => {
    dispatch(imageToTokenizeChanged(null));
  }, [dispatch]);
  useHotkeys('esc', exitTokenization, [exitTokenization]);

  return (
    <Flex w="full" justifyContent="center" h={8}>
      <Flex flex={1} justifyContent="center">
        <ButtonGroup size="sm" variant="outline" alignItems="center">
          <Button
            flexShrink={0}
            onClick={setTokenizationDisplayPositivePrompt}
            colorScheme={tokenizationPrompt === 'positive' ? 'invokeBlue' : 'base'}
          >
            {t('gallery.tokenizationPositive')}
          </Button>
          <Button
            flexShrink={0}
            onClick={setTokenizationDisplayNegativePrompt}
            colorScheme={tokenizationPrompt === 'negative' ? 'invokeBlue' : 'base'}
          >
            {t('gallery.tokenizationNegative')}
          </Button>
        </ButtonGroup>
      </Flex>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineStart="auto" alignItems="center">
          <Tooltip label={<TokenizationHelp />}>
            <Flex alignItems="center">
              <Icon boxSize={6} color="base.300" as={PiQuestion} lineHeight={0} />
            </Flex>
          </Tooltip>
          <Button
            size="sm"
            variant="link"
            alignSelf="stretch"
            px={2}
            aria-label={`${t('gallery.exitTokenization')} (Esc)`}
            tooltip={`${t('gallery.exitTokenization')} (Esc)`}
            onClick={exitTokenization}
          >
            {t('gallery.exitTokenization')}
          </Button>
        </Flex>
      </Flex>
    </Flex>
  );
});

TokenizationToolbar.displayName = 'TokenizationToolbar';

const TokenizationHelp = () => {
  return (
    <Trans i18nKey="gallery.tokenizationHelp" components={{ Kbd: <Kbd /> }}></Trans>
  );
};
