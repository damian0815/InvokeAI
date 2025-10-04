import {
  Button,
  ButtonGroup,
  Flex,
  Icon,
  IconButton,
  Kbd,
  ListItem,
  Tooltip,
  UnorderedList,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { imageToTokenizeChanged } from 'features/gallery/store/gallerySlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { Trans, useTranslation } from 'react-i18next';
import { PiArrowsLeftRightBold, PiArrowsOutBold, PiQuestion } from 'react-icons/pi';

export const TokenizationToolbar = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const exitTokenization = useCallback(() => {
    dispatch(imageToTokenizeChanged(null));
  }, [dispatch]);
  useHotkeys('esc', exitTokenization, [exitTokenization]);

  return (
    <Flex w="full" justifyContent="center" h={8}>
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
